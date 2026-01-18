import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def generate_boutiques_descriptor(module, interface_name, container_image, container_type, container_index=None, verbose=False, save=False, save_path=None, author=None, ignore_inputs=None, tags=None):
    """
    Generate a JSON Boutiques description of a Nipype interface.

    Arguments
    ---------
    module :
        module where the Nipype interface is declared.
    interface_name :
        name of Nipype interface.
    container_image :
        name of the container image where the tool is installed
    container_type :
        type of container image (Docker or Singularity)
    container_index :
        optional index where the image is available
    verbose :
        print information messages
    save :
        True if you want to save descriptor to a file
    save_path :
        file path for the saved descriptor (defaults to name of the
      interface in current directory)
    author :
        author of the tool (required for publishing)
    ignore_inputs :
        list of interface inputs to not include in the descriptor
    tags :
        JSON object containing tags to include in the descriptor,
        e.g. ``{"key1": "value1"}`` (note: the tags 'domain:neuroinformatics'
        and 'interface-type:nipype' are included by default)

    Returns
    -------
    boutiques : str
       string containing a Boutiques' JSON object

    """
    if not module:
        raise Exception('Undefined module.')
    if isinstance(module, (str, bytes)):
        import_module(module)
        module_name = str(module)
        module = sys.modules[module]
    else:
        module_name = str(module.__name__)
    interface = getattr(module, interface_name)()
    inputs = interface.input_spec()
    outputs = interface.output_spec()
    tool_desc = {}
    tool_desc['name'] = interface_name
    tool_desc['command-line'] = interface_name + ' '
    tool_desc['author'] = 'Nipype (interface)'
    if author is not None:
        tool_desc['author'] = tool_desc['author'] + ', ' + author + ' (tool)'
    tool_desc['description'] = interface_name + ', as implemented in Nipype (module: ' + module_name + ', interface: ' + interface_name + ').'
    tool_desc['inputs'] = []
    tool_desc['output-files'] = []
    tool_desc['groups'] = []
    tool_desc['tool-version'] = interface.version if interface.version is not None else '1.0.0'
    tool_desc['schema-version'] = '0.5'
    if container_image:
        tool_desc['container-image'] = {}
        tool_desc['container-image']['image'] = container_image
        tool_desc['container-image']['type'] = container_type
        if container_index:
            tool_desc['container-image']['index'] = container_index
    for name, spec in sorted(interface.inputs.traits(transient=None).items()):
        if ignore_inputs is not None and name in ignore_inputs:
            continue
        elif spec.name_source and spec.name_template:
            tool_desc['output-files'].append(get_boutiques_output_from_inp(inputs, spec, name))
        else:
            inp = get_boutiques_input(inputs, interface, name, spec, verbose)
            if isinstance(inp, list):
                mutex_group_members = []
                tool_desc['command-line'] += inp[0]['value-key'] + ' '
                for i in inp:
                    tool_desc['inputs'].append(i)
                    mutex_group_members.append(i['id'])
                    if verbose:
                        print('-> Adding input ' + i['name'])
                tool_desc['groups'].append({'id': inp[0]['id'] + '_group', 'name': inp[0]['name'] + ' group', 'members': mutex_group_members, 'mutually-exclusive': True})
            else:
                tool_desc['inputs'].append(inp)
                tool_desc['command-line'] += inp['value-key'] + ' '
                if verbose:
                    print('-> Adding input ' + inp['name'])
    tool_desc['groups'] += get_boutiques_groups(interface.inputs.traits(transient=None).items())
    if len(tool_desc['groups']) == 0:
        del tool_desc['groups']
    generate_tool_outputs(outputs, interface, tool_desc, verbose, True)
    custom_inputs = generate_custom_inputs(tool_desc['inputs'])
    for input_dict in custom_inputs:
        interface = getattr(module, interface_name)(**input_dict)
        outputs = interface.output_spec()
        generate_tool_outputs(outputs, interface, tool_desc, verbose, False)
    for output in tool_desc['output-files']:
        if output['path-template'] == '':
            fill_in_missing_output_path(output, output['name'], tool_desc['inputs'])
    desc_tags = {'domain': 'neuroinformatics', 'source': 'nipype-interface'}
    if tags is not None:
        tags_dict = json.loads(tags)
        for k, v in tags_dict.items():
            if k in desc_tags:
                if not isinstance(desc_tags[k], list):
                    desc_tags[k] = [desc_tags[k]]
                desc_tags[k].append(v)
            else:
                desc_tags[k] = v
    tool_desc['tags'] = desc_tags
    tool_desc['command-line'] = reorder_cmd_line_args(tool_desc['command-line'], interface, ignore_inputs)
    tool_desc['command-line'] = tool_desc['command-line'].strip()
    if save:
        path = save_path or os.path.join(os.getcwd(), interface_name + '.json')
        with open(path, 'w') as outfile:
            json.dump(tool_desc, outfile, indent=4, separators=(',', ': '))
        if verbose:
            print('-> Descriptor saved to file ' + outfile.name)
    print('NOTE: Descriptors produced by this script may not entirely conform to the Nipype interface specs. Please check that the descriptor is correct before using it.')
    return json.dumps(tool_desc, indent=4, separators=(',', ': '))