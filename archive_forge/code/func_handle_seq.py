import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
def handle_seq(seq, name):
    tmp = {}
    if isinstance(seq, list):
        tmp = []
        for _ in seq:
            cls = build_class(name.capitalize(), _)
            tmp.append(cls)
        return tmp
    for k, v in seq.items():
        if isinstance(v, MutableSequence):
            for _ in v:
                if isinstance(_, Mapping):
                    types = [type(x) for x in _.values()]
                    if XmlDictConfig in types:
                        result = handle_map(_, k)
                        if isinstance(result, tuple):
                            tmp.update({result[0]: result[1]})
                        else:
                            tmp.update({k: result})
                    else:
                        tmp_list = [build_class(k.capitalize(), i) for i in v]
                        tmp[k] = tmp_list
        elif isinstance(v, str):
            tmp.update({k: v})
    return tmp