from dataclasses import dataclass
from typing import List, Tuple
from langchain_core.utils.json_schema import dereference_refs
def reduce_openapi_spec(spec: dict, dereference: bool=True) -> ReducedOpenAPISpec:
    """Simplify/distill/minify a spec somehow.

    I want a smaller target for retrieval and (more importantly)
    I want smaller results from retrieval.
    I was hoping https://openapi.tools/ would have some useful bits
    to this end, but doesn't seem so.
    """
    endpoints = [(f'{operation_name.upper()} {route}', docs.get('description'), docs) for route, operation in spec['paths'].items() for operation_name, docs in operation.items() if operation_name in ['get', 'post', 'patch', 'put', 'delete']]
    if dereference:
        endpoints = [(name, description, dereference_refs(docs, full_schema=spec)) for name, description, docs in endpoints]

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
    endpoints = [(name, description, reduce_endpoint_docs(docs)) for name, description, docs in endpoints]
    return ReducedOpenAPISpec(servers=spec['servers'], description=spec['info'].get('description', ''), endpoints=endpoints)