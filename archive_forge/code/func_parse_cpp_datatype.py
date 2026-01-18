from ..common.utils import bytes2str
def parse_cpp_datatype(var_die):
    """Given a DIE that describes a variable, a parameter, or a member
    with DW_AT_type in it, tries to return the C++ datatype as a string

    Returns a TypeDesc.

    Does not follow typedefs, doesn't  resolve array element types
    or struct members. Not good for a debugger.
    """
    t = TypeDesc()
    if not 'DW_AT_type' in var_die.attributes:
        t.tag = ''
        return t
    type_die = var_die.get_DIE_from_attribute('DW_AT_type')
    mods = []
    while type_die.tag in ('DW_TAG_const_type', 'DW_TAG_pointer_type', 'DW_TAG_reference_type'):
        modifier = _strip_type_tag(type_die)
        mods.insert(0, modifier)
        if not 'DW_AT_type' in type_die.attributes:
            t.name = t.tag = 'void'
            t.modifiers = tuple(mods)
            return t
        type_die = type_die.get_DIE_from_attribute('DW_AT_type')
    t.tag = _strip_type_tag(type_die)
    t.modifiers = tuple(mods)
    if t.tag in ('ptr_to_member', 'subroutine'):
        if t.tag == 'ptr_to_member':
            ptr_prefix = DIE_name(type_die.get_DIE_from_attribute('DW_AT_containing_type')) + '::'
            type_die = type_die.get_DIE_from_attribute('DW_AT_type')
        elif 'DW_AT_object_pointer' in type_die.attributes:
            ptr_prefix = DIE_name(DIE_type(DIE_type(type_die.get_DIE_from_attribute('DW_AT_object_pointer')))) + '::'
        else:
            ptr_prefix = ''
        if t.tag == 'subroutine':
            params = tuple((format_function_param(p, p) for p in type_die.iter_children() if p.tag in ('DW_TAG_formal_parameter', 'DW_TAG_unspecified_parameters') and 'DW_AT_artificial' not in p.attributes))
            params = ', '.join(params)
            if 'DW_AT_type' in type_die.attributes:
                retval_type = parse_cpp_datatype(type_die)
                is_pointer = retval_type.modifiers and retval_type.modifiers[-1] == 'pointer'
                retval_type = str(retval_type)
                if not is_pointer:
                    retval_type += ' '
            else:
                retval_type = 'void '
            if len(mods) and mods[-1] == 'pointer':
                mods.pop()
                t.modifiers = tuple(mods)
                t.name = '%s(%s*)(%s)' % (retval_type, ptr_prefix, params)
            else:
                t.name = '%s(%s)' % (retval_type, params)
            return t
    elif DIE_is_ptr_to_member_struct(type_die):
        dt = parse_cpp_datatype(next(type_die.iter_children()))
        dt.modifiers = tuple(dt.modifiers[:-1])
        dt.tag = 'ptr_to_member_type'
        return dt
    elif t.tag == 'array':
        t.dimensions = (_array_subtype_size(sub) for sub in type_die.iter_children() if sub.tag == 'DW_TAG_subrange_type')
        t.name = describe_cpp_datatype(type_die)
        return t
    t.name = safe_DIE_name(type_die, t.tag + ' ')
    parent = type_die.get_parent()
    scopes = list()
    while parent.tag in ('DW_TAG_class_type', 'DW_TAG_structure_type', 'DW_TAG_union_type', 'DW_TAG_namespace'):
        scopes.insert(0, safe_DIE_name(parent, _strip_type_tag(parent) + ' '))
        parent = parent.get_parent()
    t.scopes = tuple(scopes)
    return t