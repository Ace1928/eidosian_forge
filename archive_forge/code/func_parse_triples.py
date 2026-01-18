from __future__ import annotations
from typing import Any, List, NamedTuple, Optional, Tuple
def parse_triples(knowledge_str: str) -> List[KnowledgeTriple]:
    """Parse knowledge triples from the knowledge string."""
    knowledge_str = knowledge_str.strip()
    if not knowledge_str or knowledge_str == 'NONE':
        return []
    triple_strs = knowledge_str.split(KG_TRIPLE_DELIMITER)
    results = []
    for triple_str in triple_strs:
        try:
            kg_triple = KnowledgeTriple.from_string(triple_str)
        except ValueError:
            continue
        results.append(kg_triple)
    return results