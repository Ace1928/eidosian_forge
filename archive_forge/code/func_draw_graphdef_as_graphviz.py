import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
def draw_graphdef_as_graphviz(graphdef, dot_output_filename):
    """Exports a GraphDef to GraphViz format.

  - Step 1: Drawing Each Node of the compute GraphDef.
  - Step 2: Create nodes for each collected dtype in the graph.
  - Step 3: Creating invisible links to align properly the legend.

  Each node consequently mentions:
  - Op Type
  - Compute Dtype
  - Compute Device
  """
    dtype_index = DTypeIndex()
    with open(dot_output_filename, 'w') as f:
        print('digraph tftrt_converted_graph {', file=f)
        print('  graph [fontsize=10 fontname="Verdana"];', file=f)
        print('  node [style=filled height=0.55 colorscheme=set312 shape=box];', file=f)
        print('\n  subgraph tensorflow_graph {', file=f)
        print('    node [width=1.35];', file=f)
        nodes_with_no_inputs = []
        for node in graphdef.node:
            output_name = node.name
            node_precision = get_node_compute_dtype(node)
            color_idx = dtype_index.get_dtype_index(node_precision)
            device_key = node.device.split('/')[-1]
            if not device_key:
                device_key = 'device:Unspecified'
            if node.op == 'TRTEngineOp':
                node_count, _ = get_trtengineop_node_op_count(graphdef, output_name)
                node_label = f'{output_name} [{node_count}]'
            else:
                node_label = f'{node.op}'
            node_label = f'<b>{node_label}</b>  <br/><i>{device_key}</i>'
            print(f'    "{output_name}" [label=<{node_label}> fillcolor={color_idx}];', file=f)
            if len(node.input):
                for input_full_name in node.input:
                    parts = input_full_name.split(':')
                    input_name = re.sub('^\\^', '', parts[0])
                    print(f'  "{input_name}" -> "{output_name}";', file=f)
            else:
                nodes_with_no_inputs.append(output_name)
        print('  }', file=f)
        print('\n  subgraph cluster_legend {', file=f)
        print('    label="Compute Dtype Legend";', file=f)
        print('    margin="30";', file=f)
        print('    node [width=2];', file=f)
        for dtype, color_idx in dtype_index.items():
            print(f'    {dtype} [fillcolor={color_idx} label=<<b>{dtype}</b>>];', file=f)
        print('  }', file=f)
        print('\n  edge[style="invisible", dir="none"];', file=f)
        for dtype in dtype_index.keys():
            for node_name in nodes_with_no_inputs:
                print(f'  "{dtype}" -> "{node_name}"', file=f)
        print('}', file=f)
    print('\n===================================================================')
    print(f'Graph Visualization Exported to: `{dot_output_filename}`.')
    print('We recommend using https://edotor.net/ to visualize the .dot file.')
    print('You can also use `graphviz` utility to convert them to PNG format:')
    print('  - `sudo apt install -y graphviz`')
    print('  - `dot -Tpng <input_filename>.dot -o <output_filename>.png`')
    print('===================================================================\n')