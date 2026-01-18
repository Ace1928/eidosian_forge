import logging
import re
from argparse import (
from collections import defaultdict
from functools import total_ordering
from itertools import starmap
from string import Template
from typing import Any, Dict, List
from typing import Optional as Opt
from typing import Union
@mark_completer('zsh')
def complete_zsh(parser, root_prefix=None, preamble='', choice_functions=None):
    """
    Returns zsh syntax autocompletion script.

    See `complete` for arguments.
    """
    prog = parser.prog
    root_prefix = wordify(f'_shtab_{root_prefix or prog}')
    choice_type2fn = {k: v['zsh'] for k, v in CHOICE_FUNCTIONS.items()}
    if choice_functions:
        choice_type2fn.update(choice_functions)

    def is_opt_end(opt):
        return isinstance(opt, OPTION_END) or opt.nargs == REMAINDER

    def is_opt_multiline(opt):
        return isinstance(opt, OPTION_MULTI)

    def format_optional(opt, parser):
        get_help = parser._get_formatter()._expand_help
        return ('{nargs}{options}"[{help}]"' if isinstance(opt, FLAG_OPTION) else '{nargs}{options}"[{help}]:{dest}:{pattern}"').format(nargs='"(- : *)"' if is_opt_end(opt) else '"*"' if is_opt_multiline(opt) else '', options='{{{}}}'.format(','.join(opt.option_strings)) if len(opt.option_strings) > 1 else '"{}"'.format(''.join(opt.option_strings)), help=escape_zsh(get_help(opt) if opt.help else ''), dest=opt.dest, pattern=complete2pattern(opt.complete, 'zsh', choice_type2fn) if hasattr(opt, 'complete') else (choice_type2fn[opt.choices[0].type] if isinstance(opt.choices[0], Choice) else '({})'.format(' '.join(map(str, opt.choices)))) if opt.choices else '').replace('""', '')

    def format_positional(opt, parser):
        get_help = parser._get_formatter()._expand_help
        return '"{nargs}:{help}:{pattern}"'.format(nargs={ONE_OR_MORE: '(*)', ZERO_OR_MORE: '(*):', REMAINDER: '(-)*'}.get(opt.nargs, ''), help=escape_zsh((get_help(opt) if opt.help else opt.dest).strip().split('\n')[0]), pattern=complete2pattern(opt.complete, 'zsh', choice_type2fn) if hasattr(opt, 'complete') else (choice_type2fn[opt.choices[0].type] if isinstance(opt.choices[0], Choice) else '({})'.format(' '.join(map(str, opt.choices)))) if opt.choices else '')
    all_commands = {root_prefix: {'cmd': prog, 'arguments': [format_optional(opt, parser) for opt in parser._get_optional_actions() if opt.help != SUPPRESS] + [format_positional(opt, parser) for opt in parser._get_positional_actions() if opt.help != SUPPRESS and opt.choices is None], 'help': (parser.description or '').strip().split('\n')[0], 'commands': [], 'paths': []}}

    def recurse(parser, prefix, paths=None):
        paths = paths or []
        subcmds = []
        for sub in parser._get_positional_actions():
            if sub.help == SUPPRESS or not sub.choices:
                continue
            if not sub.choices or not isinstance(sub.choices, dict):
                all_commands[prefix]['arguments'].append(format_positional(sub, parser))
            else:
                log.debug(f'choices:{prefix}:{sorted(sub.choices)}')
                public_cmds = get_public_subcommands(sub)
                for cmd, subparser in sub.choices.items():
                    if cmd not in public_cmds:
                        log.debug('skip:subcommand:%s', cmd)
                        continue
                    log.debug('subcommand:%s', cmd)
                    arguments = [format_optional(opt, parser) for opt in subparser._get_optional_actions() if opt.help != SUPPRESS]
                    arguments.extend((format_positional(opt, parser) for opt in subparser._get_positional_actions() if not isinstance(opt.choices, dict) if opt.help != SUPPRESS))
                    formatter = subparser._get_formatter()
                    backup_width = formatter._width
                    formatter._width = 1234567
                    desc = formatter._format_text(subparser.description or '').strip()
                    formatter._width = backup_width
                    new_pref = f'{prefix}_{wordify(cmd)}'
                    options = all_commands[new_pref] = {'cmd': cmd, 'help': desc.split('\n')[0], 'arguments': arguments, 'paths': [*paths, cmd]}
                    new_subcmds = recurse(subparser, new_pref, [*paths, cmd])
                    options['commands'] = {all_commands[pref]['cmd']: all_commands[pref] for pref in new_subcmds if pref in all_commands}
                    subcmds.extend([*new_subcmds, new_pref])
                    log.debug('subcommands:%s:%s', cmd, options)
        return subcmds
    recurse(parser, root_prefix)
    all_commands[root_prefix]['commands'] = {options['cmd']: options for prefix, options in sorted(all_commands.items()) if len(options.get('paths', [])) < 2 and prefix != root_prefix}
    subcommands = {prefix: options for prefix, options in all_commands.items() if options.get('commands')}
    subcommands.setdefault(root_prefix, all_commands[root_prefix])
    log.debug('subcommands:%s:%s', root_prefix, sorted(all_commands))

    def command_case(prefix, options):
        name = options['cmd']
        commands = options['commands']
        case_fmt_on_no_sub = '{name}) _arguments -C -s ${prefix}_{name_wordify}_options ;;'
        case_fmt_on_sub = '{name}) {prefix}_{name_wordify} ;;'
        cases = []
        for _, options in sorted(commands.items()):
            fmt = case_fmt_on_sub if options.get('commands') else case_fmt_on_no_sub
            cases.append(fmt.format(name=options['cmd'], name_wordify=wordify(options['cmd']), prefix=prefix))
        cases = '\n\t'.expandtabs(8).join(cases)
        return f"""{prefix}() {{\n  local context state line curcontext="$curcontext" one_or_more='(-)*' remainder='(*)'\n\n  if ((${{{prefix}_options[(I)${{(q)one_or_more}}*]}} + ${{{prefix}_options[(I)${{(q)remainder}}*]}} == 0)); then  # noqa: E501\n    {prefix}_options+=(': :{prefix}_commands' '*::: :->{name}')\n  fi\n  _arguments -C -s ${prefix}_options\n\n  case $state in\n    {name})\n      words=($line[1] "${{words[@]}}")\n      (( CURRENT += 1 ))\n      curcontext="${{curcontext%:*:*}}:{prefix}-$line[1]:"\n      case $line[1] in\n        {cases}\n      esac\n  esac\n}}\n"""

    def command_option(prefix, options):
        arguments = '\n  '.join(options['arguments'])
        return f'{prefix}_options=(\n  {arguments}\n)\n'

    def command_list(prefix, options):
        name = ' '.join([prog, *options['paths']])
        commands = '\n    '.join((f'"{escape_zsh(cmd)}:{escape_zsh(opt['help'])}"' for cmd, opt in sorted(options['commands'].items())))
        return f"\n{prefix}_commands() {{\n  local _commands=(\n    {commands}\n  )\n  _describe '{name} commands' _commands\n}}"
    preamble = f'# Custom Preamble\n{preamble.rstrip()}\n\n# End Custom Preamble\n' if preamble else ''
    return Template('#compdef ${prog}\n\n# AUTOMATICALLY GENERATED by `shtab`\n\n${command_commands}\n\n${command_options}\n\n${command_cases}\n${preamble}\n\ntypeset -A opt_args\n\nif [[ $zsh_eval_context[-1] == eval ]]; then\n  # eval/source/. command, register function for later\n  compdef ${root_prefix} -N ${prog}\nelse\n  # autoload from fpath, call function directly\n  ${root_prefix} "$@"\nfi\n').safe_substitute(prog=prog, root_prefix=root_prefix, command_cases='\n'.join(starmap(command_case, sorted(subcommands.items()))), command_commands='\n'.join(starmap(command_list, sorted(subcommands.items()))), command_options='\n'.join(starmap(command_option, sorted(all_commands.items()))), preamble=preamble)