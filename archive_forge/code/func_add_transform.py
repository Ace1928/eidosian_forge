from docutils import languages, ApplicationError, TransformSpec
def add_transform(self, transform_class, priority=None, **kwargs):
    """
        Store a single transform.  Use `priority` to override the default.
        `kwargs` is a dictionary whose contents are passed as keyword
        arguments to the `apply` method of the transform.  This can be used to
        pass application-specific data to the transform instance.
        """
    if priority is None:
        priority = transform_class.default_priority
    priority_string = self.get_priority_string(priority)
    self.transforms.append((priority_string, transform_class, None, kwargs))
    self.sorted = 0