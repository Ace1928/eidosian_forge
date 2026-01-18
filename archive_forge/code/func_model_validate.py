import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
@classmethod
def model_validate(cls, data):
    d = deepcopy(data)
    obj = cls(**d)
    inline_comments = []
    for k, v in d.items():
        if k.startswith('inlineComment'):
            comment = InlineComment.model_validate(v)
            inline_comments.append(comment)
    obj.inline_comments = inline_comments
    return obj