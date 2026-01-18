from collections import namedtuple
import json
import logging
import pprint
import re
def command_path(self):
    if self.parent_cmd:
        return self.parent_cmd.command_path() + [self.command]
    else:
        return [self.command]