from __future__ import annotations
import dataclasses
import enum
import logging
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple
from torch.onnx._internal.diagnostics.infra import formatter, sarif
@classmethod
def from_sarif(cls, **kwargs):
    """Returns a rule from the SARIF reporting descriptor."""
    short_description = kwargs.get('short_description', {}).get('text')
    full_description = kwargs.get('full_description', {}).get('text')
    full_description_markdown = kwargs.get('full_description', {}).get('markdown')
    help_uri = kwargs.get('help_uri')
    rule = cls(id=kwargs['id'], name=kwargs['name'], message_default_template=kwargs['message_strings']['default']['text'], short_description=short_description, full_description=full_description, full_description_markdown=full_description_markdown, help_uri=help_uri)
    return rule