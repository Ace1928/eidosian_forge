from unittest import TestCase
import ibis
from fugue import ExecutionEngine, FugueWorkflow, register_default_sql_engine
from fugue_ibis import IbisEngine, as_fugue, as_ibis, run_ibis
@property
def ibis_engine(self) -> ExecutionEngine:
    return self._ibis_engine