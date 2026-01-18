import uuid
import json
import typing
import codecs
import hashlib
import datetime
import contextlib
import dataclasses
from enum import Enum
from .lazy import lazy_import, get_obj_class_name
def parse_list_str(line: typing.Optional[typing.Union[typing.List[str], str]], default: typing.Optional[typing.List[str]]=None, seperators: typing.Optional[typing.List[str]]=None) -> typing.Optional[typing.List[str]]:
    """
    Try to parse a string as a list of strings

    Args:
        line (typing.Optional[typing.Union[typing.List[str], str]]): [description]
        default (typing.Optional[typing.List[str]], optional): [description]. Defaults to None.
        seperators (typing.Optional[typing.List[str]], optional): [description]. Defaults to None.
    
    """
    if line is None:
        return default
    if seperators is None:
        seperators = [',', '|', ';']
    if isinstance(line, list):
        return line
    if isinstance(line, str):
        if '[' in line and ']' in line:
            if '"' in line or "'" in line:
                try:
                    line = json.loads(line)
                    return line
                except Exception:
                    line = line.replace("'", '').replace('"', '')
            line = line.replace('[', '').replace(']', '')
        for seperator in seperators:
            if seperator in line:
                return line.split(seperator)
        line = [line]
    return line