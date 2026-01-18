from __future__ import annotations
import re
import json
import copy
import contextlib
import operator
from abc import ABC
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from lazyops.utils import logger
from lazyops.utils.lazy import lazy_import
from lazyops.libs.fastapi_utils.types.user_roles import UserRole
def populate_extra_schemas(self):
    """
        Populate the extra schemas
        """
    if self.extra_schemas_populated:
        return
    if not self.extra_schemas:
        return
    self.extra_schemas_data = {}
    for schema in self.extra_schemas:
        if isinstance(schema, str):
            try:
                schema = lazy_import(schema)
            except Exception as e:
                logger.warning(f'Invalid Extra Schema: {schema}, {e}')
                continue
        if isinstance(schema, type(BaseModel)):
            schema_name = schema.__name__
            try:
                schema = schema.model_json_schema(ref_template=self.extra_schema_ref_template)
            except Exception as e:
                logger.warning(f'Invalid Extra Schema: {schema}, {e}')
                continue
        elif isinstance(schema, dict):
            if 'title' not in schema:
                logger.warning(f'Invalid Extra Schema. Does not contain `title` in schema: {schema}')
                continue
            schema_name = schema['title']
        else:
            logger.warning(f'Invalid Extra Schema: {schema}')
            continue
        if self.extra_schema_name_mapping and schema_name in self.extra_schema_name_mapping:
            schema['title'] = self.extra_schema_name_mapping[schema_name]
        elif self.extra_schema_prefix:
            schema['title'] = f'{self.extra_schema_prefix}{schema_name}'
        if self.extra_schema_example_callable:
            if (schema_example := self.extra_schema_example_callable(schema=schema, schema_name=schema_name)):
                schema['example'] = schema_example
        elif self.extra_schema_example_mapping and schema_name in self.extra_schema_example_mapping:
            schema['example'] = self.extra_schema_example_mapping[schema_name]
        self.extra_schemas_data[schema_name] = schema
    self.extra_schemas_populated = True