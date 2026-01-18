from typing import Any, Optional
from langchain_core.runnables.graph import Graph, LabelsDict
def get_edge_label(self, label: str) -> str:
    label = self.labels.get('edges', {}).get(label, label)
    return f'<<U>{label}</U>>'