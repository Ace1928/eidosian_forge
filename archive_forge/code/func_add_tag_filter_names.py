from oslo_versionedobjects import fields as obj_fields
def add_tag_filter_names(cls):
    """Add tag filter names to the said class.

    :param cls: The class to add tag filter names to.
    :returns: None.
    """
    cls.add_extra_filter_name('tags')
    cls.add_extra_filter_name('not-tags')
    cls.add_extra_filter_name('tags-any')
    cls.add_extra_filter_name('not-tags-any')