from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Type, Union, Tuple
from ..aliases import NamespacesType, BaseXsdType
from .default import ElementData, XMLSchemaConverter
def element_decode(self, data: ElementData, xsd_element: 'XsdElement', xsd_type: Optional[BaseXsdType]=None, level: int=0) -> Any:
    xsd_type = xsd_type or xsd_element.type
    dict_class = self.dict
    tag = self.map_qname(data.tag)
    has_local_root = not self and (not self.strip_namespaces)
    result_dict = dict_class([t for t in self.map_attributes(data.attributes)])
    if has_local_root:
        result_dict['@xmlns'] = dict_class()
    xsd_group = xsd_type.model_group
    if xsd_group is None:
        if data.text is not None:
            result_dict['$'] = data.text
    elif not data.content:
        if data.text is not None:
            result_dict['$1'] = data.text
    else:
        has_single_group = xsd_group.is_single()
        for name, value, xsd_child in self.map_content(data.content):
            try:
                if '@xmlns' in value:
                    self.transfer(value['@xmlns'])
                    if not value['@xmlns']:
                        del value['@xmlns']
                    elif '' in value['@xmlns']:
                        value['@xmlns']['$'] = value['@xmlns'].pop('')
                elif '@xmlns' in value[name]:
                    self.transfer(value[name]['@xmlns'])
                    if not value[name]['@xmlns']:
                        del value[name]['@xmlns']
                    elif '' in value[name]['@xmlns']:
                        value[name]['@xmlns']['$'] = value[name]['@xmlns'].pop('')
                if len(value) == 1:
                    value = value[name]
            except (TypeError, KeyError):
                pass
            if value is None:
                value = self.dict()
            try:
                result = result_dict[name]
            except KeyError:
                if xsd_child is None or (has_single_group and xsd_child.is_single()):
                    result_dict[name] = value
                else:
                    result_dict[name] = self.list([value])
            else:
                if not isinstance(result, MutableSequence) or not result:
                    result_dict[name] = self.list([result, value])
                elif isinstance(result[0], MutableSequence) or not isinstance(value, MutableSequence):
                    result.append(value)
                else:
                    result_dict[name] = self.list([result, value])
    if has_local_root:
        if self:
            result_dict['@xmlns'].update(self)
            if not level:
                result_dict['@xmlns']['$'] = result_dict['@xmlns'].pop('')
        else:
            del result_dict['@xmlns']
        return dict_class([(tag, result_dict)])
    elif level:
        return dict_class([('@xmlns', dict_class(self)), (tag, result_dict)])
    else:
        return dict_class([('@xmlns', dict_class(((k if k else '$', v) for k, v in self.items()))), (tag, result_dict)])