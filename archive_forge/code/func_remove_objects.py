def remove_objects(self, item):
    base_item = self.namespace_proxy_helper.unproxy(item)
    result = self.base.remove_objects(base_item)
    return self.namespace_proxy_helper.proxy(result)