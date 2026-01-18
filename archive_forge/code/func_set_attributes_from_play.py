from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.utils.display import Display
def set_attributes_from_play(self, play):
    self.force_handlers = play.force_handlers