import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def command_case(self, command):
    case = '\t%s)\n' % '|'.join(command.aliases)
    if command.plugin:
        case += '\t\t# plugin "%s"\n' % command.plugin
    options = []
    enums = []
    for option in command.options:
        for message in option.error_messages:
            case += '\t\t# %s\n' % message
        if option.registry_keys:
            for key in option.registry_keys:
                options.append('{}={}'.format(option, key))
            enums.append('%s) optEnums=( %s ) ;;' % (option, ' '.join(option.registry_keys)))
        else:
            options.append(str(option))
    case += '\t\tcmdOpts=( %s )\n' % ' '.join(options)
    if command.fixed_words:
        fixed_words = command.fixed_words
        if isinstance(fixed_words, list):
            fixed_words = '( %s )' + ' '.join(fixed_words)
        case += '\t\tfixedWords=%s\n' % fixed_words
    if enums:
        case += '\t\tcase $curOpt in\n\t\t\t'
        case += '\n\t\t\t'.join(enums)
        case += '\n\t\tesac\n'
    case += '\t\t;;\n'
    return case