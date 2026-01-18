import logging
import stevedore
from cliff import command
@property
def escaped_name(self):
    return self.name.replace('-', '_')