import textwrap
import breezy
import breezy.commands
import breezy.help
import breezy.help_topics
from breezy.doc_generate import get_autodoc_datetime
from breezy.plugin import load_plugins
def format_command(params, cmd):
    """Provides long help for each public command"""
    subsection_header = '.SS "%s"\n' % cmd._usage()
    doc = '%s\n' % cmd.__doc__
    doc = breezy.help_topics.help_as_plain_text(cmd.help())
    doc = doc.replace('\n.', '.')
    option_str = ''
    options = cmd.options()
    if options:
        option_str = '\nOptions:\n'
        for option_name, option in sorted(options.items()):
            for name, short_name, argname, help in option.iter_switches():
                if option.is_hidden(name):
                    continue
                l = '    --' + name
                if argname is not None:
                    l += ' ' + argname
                if short_name:
                    l += ', -' + short_name
                l += (30 - len(l)) * ' ' + (help or '')
                wrapped = textwrap.fill(l, initial_indent='', subsequent_indent=30 * ' ', break_long_words=False)
                option_str += wrapped + '\n'
    aliases_str = ''
    if cmd.aliases:
        if len(cmd.aliases) > 1:
            aliases_str += '\nAliases: '
        else:
            aliases_str += '\nAlias: '
        aliases_str += ', '.join(cmd.aliases)
        aliases_str += '\n'
    see_also_str = ''
    see_also = cmd.get_see_also()
    if see_also:
        see_also_str += '\nSee also: '
        see_also_str += ', '.join(see_also)
        see_also_str += '\n'
    return subsection_header + option_str + aliases_str + see_also_str + '\n' + doc + '\n'