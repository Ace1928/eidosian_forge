import textwrap
import breezy
import breezy.commands
import breezy.help
import breezy.help_topics
from breezy.doc_generate import get_autodoc_datetime
from breezy.plugin import load_plugins
def getcommand_list(params):
    """Builds summary help for command names in manpage format"""
    brzcmd = params['brzcmd']
    output = '.SH "COMMAND OVERVIEW"\n'
    for cmd_name in command_name_list():
        cmd_object = breezy.commands.get_cmd_object(cmd_name)
        if cmd_object.hidden:
            continue
        cmd_help = cmd_object.help()
        if cmd_help:
            firstline = cmd_help.split('\n', 1)[0]
            usage = cmd_object._usage()
            tmp = '.TP\n.B "{}"\n{}\n'.format(usage, firstline)
            output = output + tmp
        else:
            raise RuntimeError("Command '%s' has no help text" % cmd_name)
    return output