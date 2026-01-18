import datetime
import json
from django.contrib.postgres import forms, lookups
from django.db import models
from django.db.backends.postgresql.psycopg_any import (
from django.db.models.functions import Cast
from django.db.models.lookups import PostgresOperatorLookup
from .utils import AttributeSetter
def process_lhs(self, compiler, connection):
    lhs, lhs_params = super().process_lhs(compiler, connection)
    if isinstance(self.lhs.output_field, models.FloatField):
        lhs = '%s::numeric' % lhs
    elif isinstance(self.lhs.output_field, models.SmallIntegerField):
        lhs = '%s::integer' % lhs
    return (lhs, lhs_params)