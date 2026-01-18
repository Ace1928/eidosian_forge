import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def script(self):
    return '#compdef brz bzr\n\n%(function_name)s ()\n{\n    local ret=1\n    local -a args\n    args+=(\n%(global-options)s\n    )\n\n    _arguments $args[@] && ret=0\n\n    return ret\n}\n\n%(function_name)s\n' % {'global-options': self.global_options(), 'function_name': self.function_name}