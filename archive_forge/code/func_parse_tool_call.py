import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
def parse_tool_call(self, text):
    """
        Parse request string. Expected format: <request><tool_name>query<call>
        """
    result = re.search(f'(?<={self.request_token}).*?(?={self.call_token})', text, re.DOTALL)
    if result is None:
        return (None, None)
    else:
        extracted_text = result.group()
    result = re.search('<(.*?)>', extracted_text)
    if result is None:
        return (None, None)
    else:
        tool = result.group(1)
    query = '>'.join(extracted_text.split('>')[1:])
    return (tool, query)