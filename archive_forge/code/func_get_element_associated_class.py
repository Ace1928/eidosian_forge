from os_win._i18n import _
from os_win import exceptions
def get_element_associated_class(conn, class_name, element_instance_id=None, element_uuid=None, fields=None):
    """Returns the objects associated to an element as a list.

    :param conn: connection to be used to execute the query
    :param class_name: object's class type name to be retrieved
    :param element_instance_id: element class InstanceID
    :param element_uuid: UUID of the element
    :param fields: specific class attributes to be retrieved
    """
    if element_instance_id:
        instance_id = element_instance_id
    elif element_uuid:
        instance_id = 'Microsoft:%s' % element_uuid
    else:
        err_msg = _('Could not get element associated class. Either element instance id or element uuid must be specified.')
        raise exceptions.WqlException(err_msg)
    fields = ', '.join(fields) if fields else '*'
    return conn.query("SELECT %(fields)s FROM %(class_name)s WHERE InstanceID LIKE '%(instance_id)s%%'" % {'fields': fields, 'class_name': class_name, 'instance_id': instance_id})