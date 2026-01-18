from parlai.core.teachers import FixedDialogTeacher
from parlai.core.message import Message
from .build import build
import os
import json
def save_messages_from_path(self, json_path):
    with open(json_path) as f:
        for line in f:
            if len(line) > 1:
                self.messages.append(json.loads(line)['dialogue'])