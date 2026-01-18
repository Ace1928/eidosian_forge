from typing import Dict, List, MutableMapping, Optional, Set, Tuple
from onnx import GraphProto, ModelProto, TensorProto, checker, helper, utils
def merge_models(m1: ModelProto, m2: ModelProto, io_map: List[Tuple[str, str]], inputs: Optional[List[str]]=None, outputs: Optional[List[str]]=None, prefix1: Optional[str]=None, prefix2: Optional[str]=None, name: Optional[str]=None, doc_string: Optional[str]=None, producer_name: Optional[str]='onnx.compose.merge_models', producer_version: Optional[str]='1.0', domain: Optional[str]='', model_version: Optional[int]=1) -> ModelProto:
    """Combines two ONNX models into a single one.

    The combined model is defined by connecting the specified set of outputs/inputs.
    Those inputs/outputs not specified in the io_map argument will remain as
    inputs/outputs of the combined model.

    Both models should have the same IR version, and same operator sets imported.

    Arguments:
        m1 (ModelProto): First model
        m2 (ModelProto): Second model
        io_map (list of pairs of string): The pairs of names [(out0, in0), (out1, in1), ...]
                                          representing outputs of the first graph and inputs of the second
                                          to be connected
        inputs (list of string): Optional list of inputs to be included in the combined graph
                                 By default, all inputs not present in the ``io_map`` argument will be
                                 included in the combined model
        outputs (list of string): Optional list of outputs to be included in the combined graph
                                  By default, all outputs not present in the ``io_map`` argument will be
                                  included in the combined model
        prefix1 (string): Optional prefix to be added to all names in m1
        prefix2 (string): Optional prefix to be added to all names in m2
        name (string): Optional name for the combined graph
                       By default, the name is g1.name and g2.name concatenated with an undescore delimiter
        doc_string (string): Optional docstring for the combined graph
                             If not provided, a default docstring with the concatenation of g1 and g2 docstrings is used
        producer_name (string): Optional producer name for the combined model. Default: 'onnx.compose'
        producer_version (string): Optional producer version for the combined model. Default: "1.0"
        domain (string): Optional domain of the combined model. Default: ""
        model_version (int): Optional version of the graph encoded. Default: 1

    Returns:
        ModelProto
    """
    if type(m1) is not ModelProto:
        raise ValueError('m1 argument is not an ONNX model')
    if type(m2) is not ModelProto:
        raise ValueError('m2 argument is not an ONNX model')
    if m1.ir_version != m2.ir_version:
        raise ValueError(f'IR version mismatch {m1.ir_version} != {m2.ir_version}. Both models should have the same IR version')
    ir_version = m1.ir_version
    opset_import_map: MutableMapping[str, int] = {}
    opset_imports = list(m1.opset_import) + list(m2.opset_import)
    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_version = opset_import_map[entry.domain]
            if entry.version != found_version:
                raise ValueError(f"Can't merge two models with different operator set ids for a given domain. Got: {m1.opset_import} and {m2.opset_import}")
        else:
            opset_import_map[entry.domain] = entry.version
    if prefix1 or prefix2:
        if prefix1:
            m1_copy = ModelProto()
            m1_copy.CopyFrom(m1)
            m1 = m1_copy
            m1 = add_prefix(m1, prefix=prefix1)
        if prefix2:
            m2_copy = ModelProto()
            m2_copy.CopyFrom(m2)
            m2 = m2_copy
            m2 = add_prefix(m2, prefix=prefix2)
        io_map = [(prefix1 + io[0] if prefix1 else io[0], prefix2 + io[1] if prefix2 else io[1]) for io in io_map]
    graph = merge_graphs(m1.graph, m2.graph, io_map, inputs=inputs, outputs=outputs, name=name, doc_string=doc_string)
    model = helper.make_model(graph, producer_name=producer_name, producer_version=producer_version, domain=domain, model_version=model_version, opset_imports=opset_imports, ir_version=ir_version)
    model_props = {}
    for meta_entry in m1.metadata_props:
        model_props[meta_entry.key] = meta_entry.value
    for meta_entry in m2.metadata_props:
        if meta_entry.key in model_props:
            value = model_props[meta_entry.key]
            if value != meta_entry.value:
                raise ValueError(f"Can't merge models with different values for the same model metadata property. Found: property = {meta_entry.key}, with values {value} and {meta_entry.value}.")
        else:
            model_props[meta_entry.key] = meta_entry.value
    helper.set_model_props(model, model_props)
    function_overlap = list({f.name for f in m1.functions} & {f.name for f in m2.functions})
    if function_overlap:
        raise ValueError("Can't merge models with overlapping local function names. Found in both graphs: " + ', '.join(function_overlap))
    model.functions.MergeFrom(m1.functions)
    model.functions.MergeFrom(m2.functions)
    checker.check_model(model)
    return model