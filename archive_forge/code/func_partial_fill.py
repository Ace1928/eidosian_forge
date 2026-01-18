import string
from typing import Any, List, Union
def partial_fill(self, **kwargs: Any) -> 'PromptTemplate':
    safe_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    new_template_strs = []
    for template_str in self.template_strs:
        extracted_variables = [fname for _, fname, _, _ in string.Formatter().parse(template_str) if fname]
        safe_available_kwargs = {k: safe_kwargs.get(k, '{' + k + '}') for k in extracted_variables}
        new_template_strs.append(template_str.format_map(safe_available_kwargs))
    return PromptTemplate(template_str=new_template_strs)