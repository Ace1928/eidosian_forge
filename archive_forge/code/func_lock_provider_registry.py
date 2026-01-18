def lock_provider_registry(self):
    super(ProviderAPIRegistry, self).__setattr__('locked', True)