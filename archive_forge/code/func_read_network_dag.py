def read_network_dag(processed_deploy_prototxt):
    """
    Reads from the caffe prototxt the network structure
    :param processed_deploy_prototxt: name of prototxt to load, preferably the prototxt should
     be processed before using a call to process_network_proto()
    :return: network_def, layer_name_to_record, top_to_layers
    network_def: caffe network structure, gives access to *all* the network information
    layer_name_to_record: *ordered* dictionary which maps between layer name and a structure which
      describes in a simple form the layer parameters
    top_to_layers: dictionary which maps a blob name to an ordered list of layers which output it
     when a top is used several times, like in inplace layhers, the list will contain all the layers
     by order of appearance
    """
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format
    from collections import OrderedDict
    network_def = caffe_pb2.NetParameter()
    with open(processed_deploy_prototxt, 'r') as proto_file:
        text_format.Merge(str(proto_file.read()), network_def)
    layer_name_to_record = OrderedDict()
    for layer_def in network_def.layer:
        if len(layer_def.include) == 0 or caffe_pb2.TEST in [item.phase for item in layer_def.include]:
            layer_name_to_record[layer_def.name] = LayerRecord(layer_def)
    top_to_layers = dict()
    for layer in network_def.layer:
        if len(layer.include) == 0 or caffe_pb2.TEST in [item.phase for item in layer.include]:
            for top in layer.top:
                if top not in top_to_layers:
                    top_to_layers[top] = list()
                top_to_layers[top].append(layer.name)
    for child_layer_name in layer_name_to_record.keys():
        child_layer_def = layer_name_to_record[child_layer_name]
        for bottom in child_layer_def.bottoms:
            if bottom in top_to_layers:
                for parent_layer_name in top_to_layers[bottom]:
                    if parent_layer_name in layer_name_to_record:
                        parent_layer_def = layer_name_to_record[parent_layer_name]
                        if parent_layer_def not in child_layer_def.parents:
                            child_layer_def.parents.append(parent_layer_def)
                        if child_layer_def not in parent_layer_def.children:
                            parent_layer_def.children.append(child_layer_def)
    for layer_name in layer_name_to_record.keys():
        layer_def = layer_name_to_record[layer_name]
        if layer_def.type == 'Eltwise' and len(layer_def.parents) == 1 and (layer_def.parents[0].type == 'Slice') and (len(layer_def.parents[0].parents) == 1) and (layer_def.parents[0].parents[0].type in ['Convolution', 'InnerProduct']):
            layer_def.filter = layer_def.parents[0].parents[0].filter
            layer_def.stride = layer_def.parents[0].parents[0].stride
            layer_def.pad = layer_def.parents[0].parents[0].pad
    return (network_def, layer_name_to_record, top_to_layers)