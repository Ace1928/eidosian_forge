def filter_to_mongo(self, filter):
    if self._is_individual(filter):
        return self._to_mongo_individual(filter)
    elif self._is_group(filter):
        return {self.GROUP_OP_TO_MONGO[filter['op']]: [self.filter_to_mongo(f) for f in filter['filters']]}