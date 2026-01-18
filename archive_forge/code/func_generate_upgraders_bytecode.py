from typing import List
from torch._C import _compile_graph_to_code_table, _generate_upgraders_graph
def generate_upgraders_bytecode() -> List:
    yaml_content = []
    upgraders_graph_map = _generate_upgraders_graph()
    for upgrader_name, upgrader_graph in upgraders_graph_map.items():
        bytecode_table = _compile_graph_to_code_table(upgrader_name, upgrader_graph)
        entry = {upgrader_name: format_bytecode(bytecode_table)}
        yaml_content.append(entry)
    return yaml_content