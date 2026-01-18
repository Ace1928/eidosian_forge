from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.helper import object_to_dict
def saveAsTable(self, name: str, format: t.Optional[str]=None, mode: t.Optional[str]=None):
    if format is not None:
        raise NotImplementedError('Providing Format in the save as table is not supported')
    exists, replace, mode = (None, None, mode or str(self._mode))
    if mode == 'append':
        return self.insertInto(name)
    if mode == 'ignore':
        exists = True
    if mode == 'overwrite':
        replace = True
    output_expression_container = exp.Create(this=exp.to_table(name), kind='TABLE', exists=exists, replace=replace)
    return self.copy(_df=self._df.copy(output_expression_container=output_expression_container))