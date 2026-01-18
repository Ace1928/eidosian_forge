from __future__ import annotations as _annotations
import dataclasses
import inspect
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import (
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Annotated, Literal, TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26
from ._internal import (
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonSchemaExtraCallable, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticUserError
def get_defs_ref(self, core_mode_ref: CoreModeRef) -> DefsRef:
    """Override this method to change the way that definitions keys are generated from a core reference.

        Args:
            core_mode_ref: The core reference.

        Returns:
            The definitions key.
        """
    core_ref, mode = core_mode_ref
    components = re.split('([\\][,])', core_ref)
    components = [x.rsplit(':', 1)[0] for x in components]
    core_ref_no_id = ''.join(components)
    components = [re.sub('(?:[^.[\\]]+\\.)+((?:[^.[\\]]+))', '\\1', x) for x in components]
    short_ref = ''.join(components)
    mode_title = _MODE_TITLE_MAPPING[mode]
    name = DefsRef(self.normalize_name(short_ref))
    name_mode = DefsRef(self.normalize_name(short_ref) + f'-{mode_title}')
    module_qualname = DefsRef(self.normalize_name(core_ref_no_id))
    module_qualname_mode = DefsRef(f'{module_qualname}-{mode_title}')
    module_qualname_id = DefsRef(self.normalize_name(core_ref))
    occurrence_index = self._collision_index.get(module_qualname_id)
    if occurrence_index is None:
        self._collision_counter[module_qualname] += 1
        occurrence_index = self._collision_index[module_qualname_id] = self._collision_counter[module_qualname]
    module_qualname_occurrence = DefsRef(f'{module_qualname}__{occurrence_index}')
    module_qualname_occurrence_mode = DefsRef(f'{module_qualname_mode}__{occurrence_index}')
    self._prioritized_defsref_choices[module_qualname_occurrence_mode] = [name, name_mode, module_qualname, module_qualname_mode, module_qualname_occurrence, module_qualname_occurrence_mode]
    return module_qualname_occurrence_mode