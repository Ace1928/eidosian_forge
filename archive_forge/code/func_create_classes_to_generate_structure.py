def create_classes_to_generate_structure(json_schema_data):
    definitions = json_schema_data['definitions']
    class_to_generatees = {}
    for name, definition in definitions.items():
        all_of = definition.get('allOf')
        description = definition.get('description')
        is_enum = definition.get('type') == 'string' and 'enum' in definition
        enum_values = None
        if is_enum:
            enum_values = definition['enum']
        properties = {}
        properties.update(definition.get('properties', {}))
        required = _OrderedSet(definition.get('required', _OrderedSet()))
        base_definitions = []
        if all_of is not None:
            for definition in all_of:
                ref = definition.get('$ref')
                if ref is not None:
                    assert ref.startswith('#/definitions/')
                    ref = ref[len('#/definitions/'):]
                    base_definitions.append(ref)
                else:
                    if not description:
                        description = definition.get('description')
                    properties.update(definition.get('properties', {}))
                    required.update(_OrderedSet(definition.get('required', _OrderedSet())))
        if isinstance(description, (list, tuple)):
            description = '\n'.join(description)
        if name == 'ModulesRequest':
            required.discard('arguments')
        class_to_generatees[name] = dict(name=name, properties=properties, base_definitions=base_definitions, description=description, required=required, is_enum=is_enum, enum_values=enum_values)
    return class_to_generatees