import csv
import datetime
import os
from collections import OrderedDict, defaultdict
from typing import IO, Any, Dict, List, Optional, Set
from tabulate import tabulate
import onnx
from onnx import GraphProto, defs, helper
def report_text(self, writer: IO[str]) -> None:
    writer.write('---------- onnx coverage: ----------\n')
    writer.write(f'Operators (passed/loaded/total): {len(self.buckets['passed'])}/{len(self.buckets['loaded'])}/{len(_all_schemas)}\n')
    writer.write('------------------------------------\n')
    rows = []
    passed = []
    all_ops: List[str] = []
    experimental: List[str] = []
    for op_cov in self.buckets['passed'].values():
        covered_attrs = [f'{attr_cov.name}: {len(attr_cov.values)}' for attr_cov in op_cov.attr_coverages.values()]
        uncovered_attrs = [f'{attr}: 0' for attr in op_cov.schema.attributes if attr not in op_cov.attr_coverages]
        attrs = sorted(covered_attrs) + sorted(uncovered_attrs)
        if attrs:
            attrs_column = os.linesep.join(attrs)
        else:
            attrs_column = 'No attributes'
        rows.append([op_cov.op_type, attrs_column])
        passed.append(op_cov.op_type)
    writer.write(tabulate(rows, headers=['Operator', 'Attributes\n(name: #values)'], tablefmt='plain'))
    writer.write('\n')
    if os.environ.get('CSVDIR') is not None:
        self.report_csv(all_ops, passed, experimental)