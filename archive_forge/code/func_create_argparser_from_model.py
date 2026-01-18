from __future__ import annotations
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Set, TYPE_CHECKING
def create_argparser_from_model(model: Type['BaseModel'], exit_on_error: Optional[bool]=False, parser_class: Optional[Type[ArgParser]]=None, **argparser_kwargs) -> ArgParser:
    """
    Create an Argument Parser from a Pydantic Model
    """
    global _parsed_parser_objects
    model_key = f'{model.__module__}.{model.__name__}'
    if model_key not in _parsed_parser_objects:
        if parser_class is None:
            parser_class = ArgParser
        doc = model.__doc__ or ''
        usage = argparser_kwargs.pop('usage', None)
        if 'Usage' in doc and (not usage):
            parts = doc.split('Usage:', 1)
            doc, usage = (parts[0].strip(), parts[-1].strip())
        parser = parser_class(description=doc, usage=usage, exit_on_error=exit_on_error, **argparser_kwargs)
        for name, field in model.model_fields.items():
            if field.json_schema_extra and field.json_schema_extra.get('hidden', False):
                continue
            args, kwargs = parse_one(name, field)
            parser.add_argument(*args, **kwargs)
        _parsed_parser_objects[model_key] = parser
    return _parsed_parser_objects[model_key]