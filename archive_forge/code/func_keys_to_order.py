def keys_to_order(self, keys):
    orders = []
    for key in keys['keys']:
        order = self.key_to_server_path(key['key'])
        if key.get('ascending'):
            order = '+' + order
        else:
            order = '-' + order
        orders.append(order)
    return orders