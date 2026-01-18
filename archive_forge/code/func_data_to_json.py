import parlai.core.build_data as build_data
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from .base_agent import _BaseSafetyTeacher
from .build import build
import copy
import json
import os
import random
import sys as _sys
def data_to_json(self, pd, file_name):
    response = pd.to_dict('records')
    with open(os.path.join(self.data_path, file_name), 'w') as f:
        f.write(json.dumps(response, indent=4))