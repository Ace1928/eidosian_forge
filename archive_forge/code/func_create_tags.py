def create_tags(self, **kwargs):
    self.meta.client.create_tags(**kwargs)
    resources = kwargs.get('Resources', [])
    tags = kwargs.get('Tags', [])
    tag_resources = []
    for resource in resources:
        for tag in tags:
            tag_resource = self.Tag(resource, tag['Key'], tag['Value'])
            tag_resources.append(tag_resource)
    return tag_resources