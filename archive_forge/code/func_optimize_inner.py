def optimize_inner(self, operations, app_label):
    """Inner optimization loop."""
    new_operations = []
    for i, operation in enumerate(operations):
        right = True
        for j, other in enumerate(operations[i + 1:]):
            result = operation.reduce(other, app_label)
            if isinstance(result, list):
                in_between = operations[i + 1:i + j + 1]
                if right:
                    new_operations.extend(in_between)
                    new_operations.extend(result)
                elif all((op.reduce(other, app_label) is True for op in in_between)):
                    new_operations.extend(result)
                    new_operations.extend(in_between)
                else:
                    new_operations.append(operation)
                    break
                new_operations.extend(operations[i + j + 2:])
                return new_operations
            elif not result:
                right = False
        else:
            new_operations.append(operation)
    return new_operations