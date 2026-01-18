from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
def convert_to_sql(self, meta_document_orm):
    return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value <= self.comparison_value)