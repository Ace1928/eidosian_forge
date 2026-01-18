from types import NoneType
from django.db.backends.utils import names_digest, split_identifier
from django.db.models.expressions import Col, ExpressionList, F, Func, OrderBy
from django.db.models.functions import Collate
from django.db.models.query_utils import Q
from django.db.models.sql import Query
from django.utils.functional import partition
def set_wrapper_classes(self, connection=None):
    if connection and connection.features.collate_as_index_expression:
        self.wrapper_classes = tuple([wrapper_cls for wrapper_cls in self.wrapper_classes if wrapper_cls is not Collate])