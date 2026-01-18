def join_fields(self, key, vals):
    if len(vals) == 1:
        return [BaseNode(f'@{key}:{vals[0].to_string()}')]
    if not vals[0].combinable:
        return [BaseNode(f'@{key}:{v.to_string()}') for v in vals]
    s = BaseNode(f'@{key}:({self.JOINSTR.join((v.to_string() for v in vals))})')
    return [s]