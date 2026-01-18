def hkdf(self, chaining_key, input_key_material, num_outputs):
    """
        HKDF(chaining_key, input_key_material, num_outputs)
        Takes a chaining_key byte sequence of length HASHLEN, and an input_key_material byte sequence with length
        either zero bytes, 32 bytes, or DHLEN bytes.
        Returns a pair or triple of byte sequences each of length HASHLEN, depending on whether num_outputs is
        two or three

        Sets temp_key = HMAC-HASH(chaining_key, input_key_material).
        Sets output1 = HMAC-HASH(temp_key, byte(0x01)).
        Sets output2 = HMAC-HASH(temp_key, output1 || byte(0x02)).
        If num_outputs == 2 then returns the pair (output1, output2).
        Sets output3 = HMAC-HASH(temp_key, output2 || byte(0x03)).
        Returns the triple (output1, output2, output3).

        :param chaining_key:
        :type chaining_key: bytes
        :param input_key_material:
        :type input_key_material: bytes
        :param num_outputs:
        :type num_outputs: int
        :return:
        :rtype: tuple
        """
    temp_key = self.hmac_hash(chaining_key, input_key_material)
    output1 = self.hmac_hash(temp_key, b'\x01')
    output2 = self.hmac_hash(temp_key, output1 + b'\x02')
    if num_outputs == 2:
        return (output1, output2)
    output3 = self.hmac_hash(temp_key, output2 + b'\x03')
    return (output1, output2, output3)