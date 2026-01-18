import logging
from botocore import xform_name
def load_rename_map(self, shape=None):
    """
        Load a name translation map given a shape. This will set
        up renamed values for any collisions, e.g. if the shape,
        an action, and a subresource all are all named ``foo``
        then the resource will have an action ``foo``, a subresource
        named ``Foo`` and a property named ``foo_attribute``.
        This is the order of precedence, from most important to
        least important:

        * Load action (resource.load)
        * Identifiers
        * Actions
        * Subresources
        * References
        * Collections
        * Waiters
        * Attributes (shape members)

        Batch actions are only exposed on collections, so do not
        get modified here. Subresources use upper camel casing, so
        are unlikely to collide with anything but other subresources.

        Creates a structure like this::

            renames = {
                ('action', 'id'): 'id_action',
                ('collection', 'id'): 'id_collection',
                ('attribute', 'id'): 'id_attribute'
            }

            # Get the final name for an action named 'id'
            name = renames.get(('action', 'id'), 'id')

        :type shape: botocore.model.Shape
        :param shape: The underlying shape for this resource.
        """
    names = set(['meta'])
    self._renamed = {}
    if self._definition.get('load'):
        names.add('load')
    for item in self._definition.get('identifiers', []):
        self._load_name_with_category(names, item['name'], 'identifier')
    for name in self._definition.get('actions', {}):
        self._load_name_with_category(names, name, 'action')
    for name, ref in self._get_has_definition().items():
        data_required = False
        for identifier in ref['resource']['identifiers']:
            if identifier['source'] == 'data':
                data_required = True
                break
        if not data_required:
            self._load_name_with_category(names, name, 'subresource', snake_case=False)
        else:
            self._load_name_with_category(names, name, 'reference')
    for name in self._definition.get('hasMany', {}):
        self._load_name_with_category(names, name, 'collection')
    for name in self._definition.get('waiters', {}):
        self._load_name_with_category(names, Waiter.PREFIX + name, 'waiter')
    if shape is not None:
        for name in shape.members.keys():
            self._load_name_with_category(names, name, 'attribute')