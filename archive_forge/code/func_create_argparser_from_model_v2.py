from __future__ import annotations
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Set, TYPE_CHECKING
def create_argparser_from_model_v2(model: Type['BaseModel'], exit_on_error: Optional[bool]=False, sub_parser: Optional[argparse.ArgumentParser]=None, **argparser_kwargs) -> ArgParser:
    """
    Create an Argument Parser from a Pydantic Model
    """
    global _parsed_parser_objects
    from pydantic import BaseModel
    model_key = f'{model.__module__}.{model.__name__}'
    if model_key not in _parsed_parser_objects:
        doc = model.__doc__ or ''
        usage = argparser_kwargs.pop('usage', None)
        if 'Usage' in doc and (not usage):
            parts = doc.split('Usage:', 1)
            doc, usage = (parts[0].strip(), parts[-1].strip())
        parser = sub_parser if sub_parser is not None else ArgParser(description=doc, usage=usage, exit_on_error=exit_on_error, **argparser_kwargs)
        if any((isinstance(field.annotation, type(BaseModel)) for field in model.model_fields.values())):
            subparsers = parser.add_subparsers(dest='subparser')
        for name, field in model.model_fields.items():
            if field.json_schema_extra and field.json_schema_extra.get('hidden', False):
                continue
            if isinstance(field.annotation, type(BaseModel)):
                subparser = subparsers.add_parser(name, help=field.description, argument_default=field.default, exit_on_error=exit_on_error)
                create_argparser_from_model_v2(field.annotation, exit_on_error=exit_on_error, sub_parser=subparser, **argparser_kwargs)
                continue
            args, kwargs = parse_one(name, field)
            parser.add_argument(*args, **kwargs)
        if sub_parser is not None:
            return parser
        _parsed_parser_objects[model_key] = parser
    return _parsed_parser_objects[model_key]