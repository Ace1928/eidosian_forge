import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
def model_dump(self, **kwargs):
    data = super().model_dump(**kwargs)
    comments = self.inline_comments
    if comments is None:
        comments = []
    for comment in comments:
        ref_id = comment.ref_id
        data[f'inlineComment_{ref_id}'] = comment.model_dump()
    data.pop('inline_comments', None)
    return data