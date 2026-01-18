from dataclasses import dataclass
from typing import List, Tuple
from langchain_core.utils.json_schema import dereference_refs
def reduce_endpoint_docs(docs: dict) -> dict:
    out = {}
    if docs.get('description'):
        out['description'] = docs.get('description')
    if docs.get('parameters'):
        out['parameters'] = [parameter for parameter in docs.get('parameters', []) if parameter.get('required')]
    if '200' in docs['responses']:
        out['responses'] = docs['responses']['200']
    if docs.get('requestBody'):
        out['requestBody'] = docs.get('requestBody')
    return out