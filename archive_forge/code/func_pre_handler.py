from __future__ import print_function
from .. import (
def pre_handler(self, cmd):
    """Hook for logic before each handler starts."""
    if self._finished:
        return
    if self.interesting_commit and cmd.name == 'commit':
        if cmd.mark == self.interesting_commit:
            print(cmd.to_string())
            self._finished = True
        return
    if cmd.name in self.parsed_params:
        fields = self.parsed_params[cmd.name]
        str = cmd.dump_str(fields, self.parsed_params, self.verbose)
        print('%s' % (str,))