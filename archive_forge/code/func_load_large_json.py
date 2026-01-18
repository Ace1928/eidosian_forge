import json
import hashlib
import logging
from typing import List, Any, Dict
import json
def load_large_json(file_path):
    with open(file_path, 'r') as file:
        json_string = file.read().strip()
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(json_string):
            while idx < len(json_string) and json_string[idx].isspace():
                idx += 1
            if idx < len(json_string):
                obj, end = decoder.raw_decode(json_string[idx:])
                yield obj
                idx += end