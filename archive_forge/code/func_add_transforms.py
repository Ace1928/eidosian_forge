from docutils import languages, ApplicationError, TransformSpec
def add_transforms(self, transform_list):
    """Store multiple transforms, with default priorities."""
    for transform_class in transform_list:
        priority_string = self.get_priority_string(transform_class.default_priority)
        self.transforms.append((priority_string, transform_class, None, {}))
    self.sorted = 0