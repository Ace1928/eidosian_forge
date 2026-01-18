import datetime
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from unittest import TestCase
from uuid import uuid4
from triad.utils.io import write_text, join
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest import raises
from triad import SerializableRLock
import fugue.api as fa
from fugue import (
from fugue.column import col
from fugue.column import functions as ff
from fugue.column import lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import (
def test_deterministic_checkpoint(self):
    self.engine.conf['fugue.workflow.checkpoint.path'] = os.path.join(self.tmpdir, 'ck')
    temp_file = os.path.join(self.tmpdir, 't.parquet')

    def mock_create(dummy: int=1) -> pd.DataFrame:
        return pd.DataFrame(np.random.rand(3, 2), columns=['a', 'b'])
    with FugueWorkflow() as dag:
        a = dag.create(mock_create)
        a.save(temp_file)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        a = dag.create(mock_create)
        b = dag.load(temp_file)
        b.assert_not_eq(a)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        a = dag.create(mock_create).strong_checkpoint()
        a.save(temp_file)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        a = dag.create(mock_create).strong_checkpoint()
        b = dag.load(temp_file)
        b.assert_not_eq(a)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        dag.create(mock_create, params=dict(dummy=2))
        a = dag.create(mock_create).deterministic_checkpoint()
        id1 = a.spec_uuid()
        a.save(temp_file)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        a = dag.create(mock_create).deterministic_checkpoint()
        b = dag.load(temp_file)
        b.assert_eq(a)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        a = dag.create(mock_create).deterministic_checkpoint(partition=PartitionSpec(num=2))
        id2 = a.spec_uuid()
        b = dag.load(temp_file)
        b.assert_eq(a)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        a = dag.create(mock_create, params={'dummy': 2}).deterministic_checkpoint()
        id3 = a.spec_uuid()
        b = dag.load(temp_file)
        b.assert_not_eq(a)
    dag.run(self.engine)
    assert id1 == id2
    assert id1 != id3