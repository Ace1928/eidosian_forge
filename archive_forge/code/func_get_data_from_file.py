from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json
import re
import csv
def get_data_from_file(self, filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data