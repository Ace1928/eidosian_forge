import csv
import datetime
import os
from collections import OrderedDict, defaultdict
from typing import IO, Any, Dict, List, Optional, Set
from tabulate import tabulate
import onnx
from onnx import GraphProto, defs, helper
def report_csv(self, all_ops: List[str], passed: List[Optional[str]], experimental: List[str]) -> None:
    for schema in _all_schemas:
        if schema.domain in {'', 'ai.onnx'}:
            all_ops.append(schema.name)
            if schema.support_level == defs.OpSchema.SupportType.EXPERIMENTAL:
                experimental.append(schema.name)
    all_ops.sort()
    nodes_path = os.path.join(str(os.environ.get('CSVDIR')), 'nodes.csv')
    models_path = os.path.join(str(os.environ.get('CSVDIR')), 'models.csv')
    existing_nodes: OrderedDict[str, Dict[str, str]] = OrderedDict()
    existing_models: OrderedDict[str, Dict[str, str]] = OrderedDict()
    frameworks: List[str] = []
    if os.path.isfile(nodes_path):
        with open(nodes_path) as nodes_file:
            reader = csv.DictReader(nodes_file)
            assert reader.fieldnames
            frameworks = list(reader.fieldnames)
            for row in reader:
                op = row['Op']
                del row['Op']
                existing_nodes[str(op)] = row
    if os.path.isfile(models_path):
        with open(models_path) as models_file:
            reader = csv.DictReader(models_file)
            for row in reader:
                model = row['Model']
                del row['Model']
                existing_models[str(model)] = row
    backend = os.environ.get('BACKEND')
    other_frameworks = frameworks[1:]
    with open(nodes_path, 'w') as nodes_file:
        if 'Op' not in frameworks:
            frameworks.append('Op')
        if backend not in frameworks:
            frameworks.append(str(backend))
        else:
            other_frameworks.remove(str(backend))
        node_writer = csv.DictWriter(nodes_file, fieldnames=frameworks)
        node_writer.writeheader()
        for node in all_ops:
            node_name = node
            if node in experimental:
                node_name = node + ' (Experimental)'
            if node_name not in existing_nodes:
                existing_nodes[node_name] = OrderedDict()
                for other_framework in other_frameworks:
                    existing_nodes[node_name][other_framework] = 'Skipped!'
            if node in passed:
                existing_nodes[node_name][str(backend)] = 'Passed!'
            else:
                existing_nodes[node_name][str(backend)] = 'Failed!'
        summaries: Dict[Any, Any] = {}
        if 'Summary' in existing_nodes:
            summaries = existing_nodes['Summary']
            del existing_nodes['Summary']
        summaries[str(backend)] = f'{len(passed)}/{len(all_ops)} node tests passed'
        summaries['Op'] = 'Summary'
        for node in existing_nodes:
            existing_nodes[node]['Op'] = str(node)
            node_writer.writerow(existing_nodes[node])
        node_writer.writerow(summaries)
    with open(models_path, 'w') as models_file:
        frameworks[0] = 'Model'
        model_writer = csv.DictWriter(models_file, fieldnames=frameworks)
        model_writer.writeheader()
        num_models = 0
        for bucket in self.models:
            for model in self.models[bucket]:
                num_covered = 0
                for node in self.models[bucket][model].node_coverages:
                    if node in passed:
                        num_covered += 1
                msg = 'Passed!'
                if bucket == 'loaded':
                    if model in self.models['passed']:
                        continue
                    msg = 'Failed!'
                num_models += 1
                if model not in existing_models:
                    existing_models[model] = OrderedDict()
                    for other_framework in other_frameworks:
                        existing_models[model][other_framework] = 'Skipped!'
                existing_models[model][str(backend)] = str(f'{num_covered}/{len(self.models[bucket][model].node_coverages)} nodes covered: {msg}')
        summaries.clear()
        if 'Summary' in existing_models:
            summaries = existing_models['Summary']
            del existing_models['Summary']
        if str(backend) in summaries:
            del summaries[str(backend)]
        summaries[str(backend)] = f'{len(self.models['passed'])}/{num_models} model tests passed'
        summaries['Model'] = 'Summary'
        for model in existing_models:
            existing_models[model]['Model'] = model
            model_writer.writerow(existing_models[model])
        model_writer.writerow(summaries)
    with open(os.path.join(str(os.environ.get('CSVDIR')), 'metadata.csv'), 'w') as metadata_file:
        metadata_writer = csv.writer(metadata_file)
        metadata_writer.writerow(['Latest Update', datetime.datetime.now().isoformat().replace('T', ' ')])