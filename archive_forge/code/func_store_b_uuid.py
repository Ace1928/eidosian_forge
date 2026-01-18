def store_b_uuid():
    global b_uuid
    b_uuid = next(iter(reality.resources_by_logical_name('B'))).uuid